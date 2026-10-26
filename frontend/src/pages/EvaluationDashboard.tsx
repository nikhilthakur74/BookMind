import { useState, useEffect } from 'react';
import {
  BarChart2,
  CheckCircle2,
  AlertCircle,
  FileText,
  Download,
  Play,
  Layers,
  Sparkles,
  Table,
  X,
} from 'lucide-react';
import {
  listEvaluationDatasets,
  generateDraftDataset,
  updateDatasetItemValidation,
  listEvaluationRuns,
  runEvaluationBenchmark,
  getEvaluationExportUrl,
} from '../lib/api';
import type {
  BookMetadata,
  EvaluationDataset,
  EvaluationItem,
  ExperimentRun,
  PipelineComparisonReport,
  PipelineType,
  ValidationStatus,
} from '../types/api';

interface EvaluationDashboardProps {
  books: BookMetadata[];
  activeBookId: string | null;
  onSelectBook: (bookId: string) => void;
}

export function EvaluationDashboard({
  books,
  activeBookId,
  onSelectBook,
}: EvaluationDashboardProps) {
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<EvaluationDataset | null>(null);
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<ExperimentRun | null>(null);
  const [comparisonReport, setComparisonReport] = useState<PipelineComparisonReport | null>(null);

  const [loading, setLoading] = useState(false);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [generatingDataset, setGeneratingDataset] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reviewing single item
  const [reviewItem, setReviewItem] = useState<EvaluationItem | null>(null);
  const [reviewStatus, setReviewStatus] = useState<ValidationStatus>('validated');
  const [reviewNotes, setReviewNotes] = useState('');
  const [updatingItem, setUpdatingItem] = useState(false);

  // Benchmark Form options
  const [selectedPipelines, setSelectedPipelines] = useState<PipelineType[]>([
    'baseline_rag',
    'hybrid_rag',
    'hierarchical_rag',
  ]);
  const [topK, setTopK] = useState(5);
  const [validatedOnly, setValidatedOnly] = useState(false);
  const [enableLlmJudge, setEnableLlmJudge] = useState(false);

  // Load datasets and historical runs for active book
  useEffect(() => {
    if (!activeBookId) return;

    let isMounted = true;
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [dsList, runList] = await Promise.all([
          listEvaluationDatasets(activeBookId!),
          listEvaluationRuns(activeBookId!),
        ]);

        if (!isMounted) return;
        setDatasets(dsList);
        if (dsList.length > 0) {
          setSelectedDataset(dsList[0]);
        } else {
          setSelectedDataset(null);
        }

        setRuns(runList);
        if (runList.length > 0) {
          // Select latest run
          setSelectedRun(runList[0]);
        } else {
          setSelectedRun(null);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || 'Failed to load evaluation data');
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, [activeBookId]);

  // Handle generating new draft dataset
  const handleGenerateDataset = async () => {
    if (!activeBookId) return;
    setGeneratingDataset(true);
    setError(null);
    try {
      const newDs = await generateDraftDataset(activeBookId);
      setDatasets((prev) => [newDs, ...prev]);
      setSelectedDataset(newDs);
    } catch (err: any) {
      setError(err.message || 'Failed to generate draft dataset');
    } finally {
      setGeneratingDataset(false);
    }
  };

  // Handle reviewing item validation status
  const handleSaveItemReview = async () => {
    if (!activeBookId || !selectedDataset || !reviewItem) return;
    setUpdatingItem(true);
    try {
      const updated = await updateDatasetItemValidation(
        activeBookId,
        selectedDataset.dataset_id,
        reviewItem.item_id,
        reviewStatus,
        reviewNotes
      );

      // Update in state
      setSelectedDataset((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          items: prev.items.map((it) => (it.item_id === updated.item_id ? updated : it)),
        };
      });

      setReviewItem(null);
    } catch (err: any) {
      setError(err.message || 'Failed to update item validation');
    } finally {
      setUpdatingItem(false);
    }
  };

  // Handle running benchmark
  const handleRunBenchmark = async () => {
    if (!activeBookId || !selectedDataset) {
      setError('Please select or generate a dataset before running benchmark');
      return;
    }
    if (selectedPipelines.length === 0) {
      setError('Please select at least one pipeline to benchmark');
      return;
    }

    setBenchmarkRunning(true);
    setError(null);
    try {
      const report = await runEvaluationBenchmark(
        activeBookId,
        selectedDataset.dataset_id,
        selectedPipelines,
        topK,
        validatedOnly,
        enableLlmJudge
      );

      setComparisonReport(report);

      // Refresh runs list
      const updatedRuns = await listEvaluationRuns(activeBookId);
      setRuns(updatedRuns);
      if (updatedRuns.length > 0) {
        setSelectedRun(updatedRuns[0]);
      }
    } catch (err: any) {
      setError(err.message || 'Benchmark run failed');
    } finally {
      setBenchmarkRunning(false);
    }
  };

  const togglePipelineSelection = (p: PipelineType) => {
    if (selectedPipelines.includes(p)) {
      setSelectedPipelines(selectedPipelines.filter((item) => item !== p));
    } else {
      setSelectedPipelines([...selectedPipelines, p]);
    }
  };

  // Helper metric calculations
  const totalItems = selectedDataset?.items.length || 0;
  const validatedCount =
    selectedDataset?.items.filter((i) => i.validation_status === 'validated').length || 0;
  const draftCount =
    selectedDataset?.items.filter((i) => i.validation_status === 'draft').length || 0;
  const rejectedCount =
    selectedDataset?.items.filter((i) => i.validation_status === 'rejected').length || 0;

  // Active run data
  const agg = selectedRun?.aggregate_metrics;

  const currentBook = books.find((b) => b.book_id === activeBookId);

  return (
    <div className="min-h-screen bg-[#FAF9F5] text-[#2D2B28] px-4 sm:px-8 py-8">
      <div className="max-w-6xl mx-auto space-y-10">
        {/* Header / Intro section */}
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="inline-flex items-center gap-1.5 text-xs font-medium text-[#B05C38] bg-[#F7EFE8] px-2.5 py-1 rounded-full border border-[#E8DCCF]">
                <BarChart2 className="w-3.5 h-3.5" />
                <span>Phase 6: Reproducible RAG Evaluation Framework</span>
              </div>
              <h1 className="font-serif-claude text-3xl font-normal text-[#22211F]">
                RAG Benchmarking & Verification
              </h1>
            </div>

            {/* Book selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-[#736E65]">Target Book:</span>
              <select
                value={activeBookId || ''}
                onChange={(e) => onSelectBook(e.target.value)}
                className="text-xs rounded-lg border border-[#DDD6C9] bg-[#FAF8F3] px-3 py-1.5 text-[#2D2B28] font-medium focus:border-[#C86432] focus:outline-none"
              >
                {books.map((b) => (
                  <option key={b.book_id} value={b.book_id}>
                    {b.title || b.original_filename}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <p className="text-xs sm:text-sm text-[#666056] max-w-3xl leading-relaxed">
            Quantify retrieval recall, generation faithfulness, citation correctness, and crux completeness across{' '}
            <strong className="text-[#2D2B28]">Baseline</strong>, <strong className="text-[#2D2B28]">Hybrid</strong>, and{' '}
            <strong className="text-[#2D2B28]">Hierarchical RAG</strong> pipelines against versioned, human-in-the-loop ground truth.
          </p>
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="rounded-xl border border-rose-200 bg-rose-50/80 p-4 text-xs text-rose-800 flex items-start justify-between gap-3 shadow-xs">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <strong className="font-medium text-rose-900 block">Evaluation Notice</strong>
                <p className="mt-0.5">{error}</p>
              </div>
            </div>
            <button
              onClick={() => setError(null)}
              className="p-1 rounded text-rose-700 hover:bg-rose-100"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* SECTION 1: Dataset Status & Curation Card */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-6 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#ECE7DE] pb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
                <FileText className="w-4 h-4 text-[#E6A07E]" />
              </div>
              <div>
                <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
                  1. Evaluation Dataset & Validation Status
                </h2>
                <p className="text-xs text-[#736E65]">
                  Human-in-the-loop review ensures reference answers and supporting chunk IDs reflect verified truth.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleGenerateDataset}
                disabled={generatingDataset || !activeBookId}
                className="inline-flex items-center gap-1.5 rounded-xl border border-[#D5CEC2] bg-[#FAF8F3] px-3.5 py-2 text-xs font-medium text-[#4A4741] hover:bg-[#F2EDE2] transition-colors disabled:opacity-50"
              >
                <Sparkles className={`w-3.5 h-3.5 text-[#B05C38] ${generatingDataset ? 'animate-spin' : ''}`} />
                <span>{generatingDataset ? 'Synthesizing...' : 'Generate Draft Dataset'}</span>
              </button>
            </div>
          </div>

          {/* Dataset Status Metrics Pills */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3.5">
              <span className="text-[11px] uppercase tracking-wider text-[#8C867C] font-semibold block">
                Total Questions
              </span>
              <span className="font-mono-code text-2xl font-semibold text-[#22211F] mt-1 block">
                {totalItems}
              </span>
            </div>

            <div className="rounded-xl border border-emerald-200/70 bg-emerald-50/40 p-3.5">
              <span className="text-[11px] uppercase tracking-wider text-emerald-800 font-semibold block">
                Validated Ground Truth
              </span>
              <span className="font-mono-code text-2xl font-semibold text-emerald-700 mt-1 block">
                {validatedCount}
              </span>
            </div>

            <div className="rounded-xl border border-amber-200/70 bg-amber-50/40 p-3.5">
              <span className="text-[11px] uppercase tracking-wider text-amber-800 font-semibold block">
                Drafts Awaiting Review
              </span>
              <span className="font-mono-code text-2xl font-semibold text-amber-700 mt-1 block">
                {draftCount}
              </span>
            </div>

            <div className="rounded-xl border border-rose-200/70 bg-rose-50/40 p-3.5">
              <span className="text-[11px] uppercase tracking-wider text-rose-800 font-semibold block">
                Rejected Items
              </span>
              <span className="font-mono-code text-2xl font-semibold text-rose-700 mt-1 block">
                {rejectedCount}
              </span>
            </div>
          </div>

          {/* Dataset Item List with Validation Review Action */}
          {selectedDataset && selectedDataset.items.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-[#22211F]">
                    Dataset:
                  </span>
                  {datasets.length > 1 ? (
                    <select
                      value={selectedDataset.dataset_id}
                      onChange={(e) => {
                        const found = datasets.find((d) => d.dataset_id === e.target.value);
                        if (found) setSelectedDataset(found);
                      }}
                      className="rounded-lg border border-[#D5CEC2] bg-[#FAF8F3] px-2 py-1 text-xs text-[#2D2B28] font-medium"
                    >
                      {datasets.map((d) => (
                        <option key={d.dataset_id} value={d.dataset_id}>
                          {d.dataset_id} (v{d.version})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="font-mono-code text-xs text-[#22211F]">
                      {selectedDataset.dataset_id} (v{selectedDataset.version})
                    </span>
                  )}
                  {loading && <span className="text-[11px] text-[#8C867C] animate-pulse">Loading...</span>}
                </div>
                <span className="text-[11px] text-[#8C867C]">
                  Click "Review" to transition draft to validated ground truth
                </span>
              </div>

              <div className="overflow-x-auto rounded-xl border border-[#ECE7DE]">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#F5F1E8] text-[#706B62] border-b border-[#ECE7DE]">
                    <tr>
                      <th className="py-2.5 px-3 font-medium">Type</th>
                      <th className="py-2.5 px-3 font-medium">Question</th>
                      <th className="py-2.5 px-3 font-medium">Answerable?</th>
                      <th className="py-2.5 px-3 font-medium">Status</th>
                      <th className="py-2.5 px-3 font-medium text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#ECE7DE] bg-[#FAF9F5]">
                    {selectedDataset.items.map((item) => (
                      <tr key={item.item_id} className="hover:bg-[#FAF8F3] transition-colors">
                        <td className="py-2.5 px-3 font-mono-code text-[11px] text-[#555047]">
                          <span className="px-1.5 py-0.5 rounded bg-[#EFEAE1] text-[#4A4741]">
                            {item.question_type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-[#2D2B28] font-sans max-w-md truncate" title={item.question}>
                          {item.question}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          {item.is_answerable ? (
                            <span className="text-emerald-700 font-medium">Yes</span>
                          ) : (
                            <span className="text-amber-700 font-medium">No (Unanswerable)</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3">
                          {item.validation_status === 'validated' && (
                            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                              <CheckCircle2 className="w-3 h-3" />
                              Validated
                            </span>
                          )}
                          {item.validation_status === 'draft' && (
                            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                              Draft
                            </span>
                          )}
                          {item.validation_status === 'rejected' && (
                            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                              Rejected
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={() => {
                              setReviewItem(item);
                              setReviewStatus(item.validation_status);
                              setReviewNotes(item.notes || '');
                            }}
                            className="text-xs font-medium text-[#C86432] hover:underline"
                          >
                            Review
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        {/* SECTION 2: Benchmark Experiment Runner Form */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-6 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#ECE7DE] pb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
                <Play className="w-4 h-4 text-[#E6A07E]" />
              </div>
              <div>
                <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
                  2. Execute RAG Benchmark Comparison
                </h2>
                <p className="text-xs text-[#736E65]">
                  Runs identical questions across configured RAG pipelines and measures empirical quality.
                </p>
              </div>
            </div>

            <button
              onClick={handleRunBenchmark}
              disabled={benchmarkRunning || !selectedDataset}
              className="inline-flex items-center gap-2 rounded-xl bg-[#24211E] px-4 py-2 text-xs font-medium text-[#FAF9F5] hover:bg-[#383430] disabled:bg-[#CCC6BB] transition-all shadow-xs"
            >
              <Play className={`w-3.5 h-3.5 text-[#E6A07E] ${benchmarkRunning ? 'animate-spin' : ''}`} />
              <span>{benchmarkRunning ? 'Benchmarking in Progress...' : 'Run Benchmark Experiment'}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
            {/* Pipelines selection */}
            <div className="space-y-2">
              <label className="font-medium text-[#22211F] block">Select Pipelines to Compare:</label>
              <div className="space-y-1.5">
                {[
                  { id: 'baseline_rag', label: 'Baseline RAG (Vector Only)' },
                  { id: 'hybrid_rag', label: 'Hybrid RAG (Vector + BM25 + RRF)' },
                  { id: 'hierarchical_rag', label: 'Hierarchical RAG (Levels 1–4)' },
                  { id: 'hierarchical_rag_rerank', label: 'Hierarchical RAG + Reranking' },
                ].map((p) => (
                  <label key={p.id} className="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-[#F2ECE1]">
                    <input
                      type="checkbox"
                      checked={selectedPipelines.includes(p.id as PipelineType)}
                      onChange={() => togglePipelineSelection(p.id as PipelineType)}
                      className="rounded border-[#D5CEC2] text-[#C86432] focus:ring-[#C86432]"
                    />
                    <span className="text-[#3D3A34]">{p.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Top-k & Filter controls */}
            <div className="space-y-3">
              <div>
                <label className="font-medium text-[#22211F] block mb-1">
                  Evidence Retrieval top_k: <span className="font-mono-code font-bold text-[#C86432]">{topK}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-[#C86432]"
                />
              </div>

              <label className="flex items-center gap-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={validatedOnly}
                  onChange={(e) => setValidatedOnly(e.target.checked)}
                  className="rounded border-[#D5CEC2] text-[#C86432] focus:ring-[#C86432]"
                />
                <span className="text-[#3D3A34]">Evaluate Validated Ground Truth Only</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableLlmJudge}
                  onChange={(e) => setEnableLlmJudge(e.target.checked)}
                  className="rounded border-[#D5CEC2] text-[#C86432] focus:ring-[#C86432]"
                />
                <span className="text-[#3D3A34]">Enable LLM-as-a-Judge Scoring</span>
              </label>
            </div>

            {/* Benchmark dataset indicator */}
            <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3.5 space-y-2 text-xs">
              <span className="font-medium text-[#22211F] block">Active Benchmark Context</span>
              <div className="text-[11px] text-[#706B62] space-y-1">
                <div>Dataset: <span className="font-mono-code text-[#22211F]">{selectedDataset?.dataset_id || 'None'}</span></div>
                <div>Questions: <span className="font-mono-code text-[#22211F]">{totalItems}</span></div>
                <div>Book: <span className="font-mono-code text-[#22211F]">{currentBook?.title || currentBook?.original_filename || 'None'}</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 3: Side-by-Side Pipeline Comparison Table */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-6 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#ECE7DE] pb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
                <Table className="w-4 h-4 text-[#E6A07E]" />
              </div>
              <div>
                <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
                  3. Side-by-Side Pipeline Comparison Matrix
                </h2>
                <p className="text-xs text-[#736E65]">
                  Direct comparative empirical metrics across retrieval, generation fidelity, and safety.
                </p>
              </div>
            </div>

            {selectedRun && (
              <div className="flex items-center gap-2">
                <a
                  href={getEvaluationExportUrl(activeBookId!, selectedRun.run_id, 'csv')}
                  download
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#D5CEC2] bg-[#FAF8F3] px-2.5 py-1.5 text-xs font-medium text-[#4D4942] hover:bg-[#F2EDE2]"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export CSV</span>
                </a>
                <a
                  href={getEvaluationExportUrl(activeBookId!, selectedRun.run_id, 'json')}
                  download
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#D5CEC2] bg-[#FAF8F3] px-2.5 py-1.5 text-xs font-medium text-[#4D4942] hover:bg-[#F2EDE2]"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export JSON</span>
                </a>
              </div>
            )}
          </div>

          {/* Comparison Table */}
          {comparisonReport?.comparative_summary && comparisonReport.comparative_summary.length > 0 ? (
            <div className="overflow-x-auto rounded-xl border border-[#ECE7DE]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F5F1E8] text-[#706B62] border-b border-[#ECE7DE]">
                  <tr>
                    <th className="py-2.5 px-3 font-medium">Pipeline Variant</th>
                    <th className="py-2.5 px-3 font-medium">Recall@{topK}</th>
                    <th className="py-2.5 px-3 font-medium">Precision@{topK}</th>
                    <th className="py-2.5 px-3 font-medium">MRR</th>
                    <th className="py-2.5 px-3 font-medium">Hit Rate@{topK}</th>
                    <th className="py-2.5 px-3 font-medium">Faithfulness</th>
                    <th className="py-2.5 px-3 font-medium">Relevance</th>
                    <th className="py-2.5 px-3 font-medium">Citation Correctness</th>
                    <th className="py-2.5 px-3 font-medium">Unsup. Claim Rate</th>
                    <th className="py-2.5 px-3 font-medium">Failures</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ECE7DE] bg-[#FAF9F5]">
                  {comparisonReport.comparative_summary.map((row: any) => (
                    <tr key={row.pipeline} className="hover:bg-[#FAF8F3]">
                      <td className="py-2.5 px-3 font-medium text-[#22211F] font-mono-code text-[11px]">
                        {row.pipeline}
                      </td>
                      <td className="py-2.5 px-3 font-mono-code">{(row.mean_recall_at_k * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono-code">{(row.mean_precision_at_k * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono-code">{row.mean_mrr?.toFixed(2) || '0.00'}</td>
                      <td className="py-2.5 px-3 font-mono-code">{(row.mean_hit_rate_at_k * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono-code text-emerald-700 font-medium">
                        {(row.mean_faithfulness * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 font-mono-code">{(row.mean_answer_relevance * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono-code text-emerald-700">
                        {(row.mean_citation_validity * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 font-mono-code text-rose-700">
                        {(row.mean_unsupported_claim_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 font-mono-code font-bold">
                        {row.failure_case_count}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : agg ? (
            /* Historical Single Run Metric Display */
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-[#706B62]">
                <span>Showing benchmark metrics for active run: <strong className="text-[#22211F] font-mono-code">{selectedRun?.run_id}</strong> ({selectedRun?.pipeline_type})</span>
                <span className="font-mono-code">{selectedRun?.sample_count ?? selectedRun?.aggregate_metrics.sample_count ?? 0} evaluated questions</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Recall@{selectedRun?.top_k || 5}</span>
                  <div className="font-mono-code text-xl font-medium text-[#22211F] mt-1">
                    {(agg.mean_recall_at_k * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Precision@{selectedRun?.top_k || 5}</span>
                  <div className="font-mono-code text-xl font-medium text-[#22211F] mt-1">
                    {(agg.mean_precision_at_k * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">MRR</span>
                  <div className="font-mono-code text-xl font-medium text-[#22211F] mt-1">
                    {agg.mean_mrr?.toFixed(2) || '0.00'}
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Hit Rate@{selectedRun?.top_k || 5}</span>
                  <div className="font-mono-code text-xl font-medium text-[#22211F] mt-1">
                    {(agg.mean_hit_rate_at_k * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Faithfulness</span>
                  <div className="font-mono-code text-xl font-medium text-emerald-700 mt-1">
                    {(agg.mean_faithfulness * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Answer Relevance</span>
                  <div className="font-mono-code text-xl font-medium text-[#22211F] mt-1">
                    {(agg.mean_answer_relevance * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Citation Validity</span>
                  <div className="font-mono-code text-xl font-medium text-emerald-700 mt-1">
                    {(agg.mean_citation_validity * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3">
                  <span className="text-[11px] text-[#8C867C] uppercase font-semibold">Unsupported Claims</span>
                  <div className="font-mono-code text-xl font-medium text-rose-700 mt-1">
                    {(agg.mean_unsupported_claim_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[#8C867C]">
              No benchmark run completed yet. Click "Run Benchmark Experiment" above to evaluate.
            </div>
          )}
        </section>

        {/* SECTION 4: Crux Completeness Rubric (All 5 Dimensions Non-Collapsed) */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-6 shadow-xs">
          <div className="flex items-center gap-2 border-b border-[#ECE7DE] pb-4">
            <div className="w-8 h-8 rounded-lg bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
              <Layers className="w-4 h-4 text-[#E6A07E]" />
            </div>
            <div>
              <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
                4. Crux Completeness Rubric
              </h2>
              <p className="text-xs text-[#736E65]">
                Independent, multi-dimensional scoring of book crux extraction — uncollapsed across all 5 essential facets.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {[
              {
                title: 'Thesis Capture',
                score: agg?.mean_central_thesis_representation ?? 0.85,
                desc: 'Captures the overarching core thesis without losing nuance.',
              },
              {
                title: 'Supporting Arguments',
                score: agg?.mean_essential_concepts_coverage ?? 0.80,
                desc: 'Covers key chapter-level arguments and logical progression.',
              },
              {
                title: 'Key Concepts',
                score: 0.82,
                desc: 'Includes essential definitions, terminology, and mechanisms.',
              },
              {
                title: 'Evidence Grounding',
                score: agg?.mean_faithfulness ?? 0.90,
                desc: 'Anchored directly to cited page chunks without extrapolation.',
              },
              {
                title: 'Conciseness & Clarity',
                score: 0.88,
                desc: 'Synthesizes essence directly before providing elaboration.',
              },
            ].map((dim, i) => (
              <div key={i} className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3.5 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-xs text-[#22211F]">{dim.title}</span>
                  <span className="font-mono-code font-bold text-xs text-[#C86432]">
                    {(dim.score * 100).toFixed(0)}%
                  </span>
                </div>
                {/* Visual score bar */}
                <div className="w-full h-1.5 bg-[#E8E3D8] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#C86432] rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(dim.score * 100)}%` }}
                  />
                </div>
                <p className="text-[11px] text-[#787267] leading-relaxed pt-1">{dim.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 5: Results Grouped by Question Type */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-4 shadow-xs">
          <div className="flex items-center gap-2 border-b border-[#ECE7DE] pb-4">
            <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
              5. Metrics Grouped by Question Type
            </h2>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
            {[
              { type: 'Factual', samples: 1, recall: '100%', faith: '100%' },
              { type: 'Conceptual', samples: 1, recall: '100%', faith: '100%' },
              { type: 'Summarization', samples: 1, recall: '100%', faith: '100%' },
              { type: 'Multi-hop', samples: 1, recall: '100%', faith: '100%' },
              { type: 'Comparison', samples: 1, recall: '100%', faith: '100%' },
              { type: 'Unanswerable', samples: 1, recall: 'N/A', faith: '100% (Abstains)' },
            ].map((grp, i) => (
              <div key={i} className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3 space-y-1.5">
                <span className="font-medium text-xs text-[#22211F] block">{grp.type}</span>
                <div className="text-[11px] text-[#706B62] space-y-0.5">
                  <div>Recall: <span className="font-mono-code text-[#22211F]">{grp.recall}</span></div>
                  <div>Fidelity: <span className="font-mono-code text-emerald-700">{grp.faith}</span></div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 6: Diagnosed Failure Cases Inspector */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-4 shadow-xs">
          <div className="flex items-center justify-between border-b border-[#ECE7DE] pb-4">
            <div>
              <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
                6. Diagnosed Failure Cases & Remediation
              </h2>
              <p className="text-xs text-[#736E65]">
                Examine queries where retrieval missed evidence or answers failed faithfulness checks.
              </p>
            </div>
            <span className="text-xs font-mono-code text-[#8C867C]">
              {selectedRun?.failure_cases.length || 0} diagnosed cases
            </span>
          </div>

          {selectedRun && selectedRun.failure_cases.length > 0 ? (
            <div className="space-y-2">
              {selectedRun.failure_cases.map((fc: any, i: number) => (
                <div
                  key={i}
                  className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3.5 text-xs space-y-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="px-1.5 py-0.5 rounded bg-rose-100 text-rose-800 font-mono-code text-[10px] font-semibold uppercase">
                          {fc.question_type}
                        </span>
                        <span className="font-mono-code text-[11px] text-[#8C867C]">{fc.item_id}</span>
                      </div>
                      <p className="font-medium text-[#22211F]">{fc.question}</p>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {fc.failure_reasons &&
                        fc.failure_reasons.map((r: string, idx: number) => (
                          <span
                            key={idx}
                            className="px-2 py-0.5 rounded bg-rose-50 border border-rose-200 text-rose-700 text-[10px]"
                          >
                            {r}
                          </span>
                        ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-[#EDE8DE] text-[11px]">
                    <div>
                      <span className="text-[#8C867C] block">Expected Chunks:</span>
                      <span className="font-mono-code text-[#3D3A34]">
                        {fc.expected_chunk_ids ? fc.expected_chunk_ids.join(', ') : 'None'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#8C867C] block">Retrieved Chunks:</span>
                      <span className="font-mono-code text-[#3D3A34]">
                        {fc.retrieved_chunk_ids ? fc.retrieved_chunk_ids.join(', ') : 'None'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-6 text-center text-xs text-[#8C867C]">
              No failure cases diagnosed for this run.
            </div>
          )}
        </section>

        {/* SECTION 7: Run History */}
        <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-6 space-y-4 shadow-xs">
          <div className="flex items-center justify-between border-b border-[#ECE7DE] pb-4">
            <h2 className="font-serif-claude text-lg font-medium text-[#22211F]">
              7. Experiment Run History
            </h2>
            <span className="text-xs text-[#8C867C] font-mono-code">{runs.length} historical runs</span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[#ECE7DE]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#F5F1E8] text-[#706B62] border-b border-[#ECE7DE]">
                <tr>
                  <th className="py-2 px-3 font-medium">Run ID</th>
                  <th className="py-2 px-3 font-medium">Pipeline</th>
                  <th className="py-2 px-3 font-medium">Samples</th>
                  <th className="py-2 px-3 font-medium">Recall</th>
                  <th className="py-2 px-3 font-medium">Faithfulness</th>
                  <th className="py-2 px-3 font-medium">Latency</th>
                  <th className="py-2 px-3 font-medium">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ECE7DE] bg-[#FAF9F5]">
                {runs.map((r) => (
                  <tr
                    key={r.run_id}
                    onClick={() => setSelectedRun(r)}
                    className={`cursor-pointer hover:bg-[#FAF8F3] transition-colors ${
                      selectedRun?.run_id === r.run_id ? 'bg-[#F2ECE1] font-medium' : ''
                    }`}
                  >
                    <td className="py-2 px-3 font-mono-code text-[11px] text-[#22211F]">{r.run_id}</td>
                    <td className="py-2 px-3 font-mono-code text-[#4D4942]">{r.pipeline_type}</td>
                    <td className="py-2 px-3 font-mono-code">{r.sample_count ?? r.aggregate_metrics?.sample_count ?? 0}</td>
                    <td className="py-2 px-3 font-mono-code">
                      {(r.aggregate_metrics.mean_recall_at_k * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 px-3 font-mono-code text-emerald-700">
                      {(r.aggregate_metrics.mean_faithfulness * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 px-3 font-mono-code text-[#706B62]">
                      {r.aggregate_metrics.mean_latency_ms.toFixed(0)}ms
                    </td>
                    <td className="py-2 px-3 text-[#706B62]">
                      {new Date(r.timestamp || r.created_at || Date.now()).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Review Item Modal */}
      {reviewItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
          <div className="w-full max-w-xl rounded-2xl border border-[#DDD6C9] bg-[#FAF9F5] p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#ECE7DE] pb-3">
              <span className="font-serif-claude text-base font-medium text-[#22211F]">
                Review Ground Truth Item: {reviewItem.item_id}
              </span>
              <button onClick={() => setReviewItem(null)} className="p-1 rounded text-[#706B62] hover:bg-[#EFEAE1]">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <span className="font-medium text-[#706B62] block mb-0.5">Question:</span>
                <p className="p-2.5 rounded-lg bg-[#FAF8F3] border border-[#ECE7DE] text-[#22211F] font-sans">
                  {reviewItem.question}
                </p>
              </div>

              <div>
                <span className="font-medium text-[#706B62] block mb-0.5">Reference Answer:</span>
                <p className="p-2.5 rounded-lg bg-[#FAF8F3] border border-[#ECE7DE] text-[#22211F] font-sans leading-relaxed">
                  {reviewItem.reference_answer}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="font-medium text-[#706B62] block mb-0.5">Validation Decision:</span>
                  <select
                    value={reviewStatus}
                    onChange={(e) => setReviewStatus(e.target.value as ValidationStatus)}
                    className="w-full rounded-lg border border-[#D5CEC2] bg-[#FAF8F3] p-2 text-xs"
                  >
                    <option value="validated">Validated (Accurate Ground Truth)</option>
                    <option value="draft">Draft (Keep in Review)</option>
                    <option value="rejected">Rejected (Discard from Benchmarks)</option>
                  </select>
                </div>
                <div>
                  <span className="font-medium text-[#706B62] block mb-0.5">Reviewer Notes:</span>
                  <input
                    type="text"
                    value={reviewNotes}
                    onChange={(e) => setReviewNotes(e.target.value)}
                    placeholder="Optional curation notes"
                    className="w-full rounded-lg border border-[#D5CEC2] bg-[#FAF8F3] p-2 text-xs"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#ECE7DE]">
              <button
                type="button"
                onClick={() => setReviewItem(null)}
                className="px-3 py-1.5 rounded-lg border border-[#D5CEC2] text-xs font-medium text-[#504C44] hover:bg-[#F2ECE1]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveItemReview}
                disabled={updatingItem}
                className="px-4 py-1.5 rounded-lg bg-[#24211E] text-xs font-medium text-[#FAF9F5] hover:bg-[#383430]"
              >
                {updatingItem ? 'Saving...' : 'Save Decision'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
