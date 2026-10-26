/**
 * Comprehensive TypeScript definitions matching all BookMind FastAPI schemas.
 */

// ==============================================================================
// Health & System
// ==============================================================================

export interface LLMHealthStatus {
  provider: string;
  model: string;
  is_reachable: boolean;
  is_model_installed: boolean;
  available_models?: string[];
  generation_successful: boolean;
  test_latency_ms?: number | null;
  error?: string | null;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'offline' | string;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  llm?: LLMHealthStatus;
}

export interface HealthState {
  data: HealthStatus | null;
  loading: boolean;
  error: string | null;
  latencyMs: number | null;
  lastChecked: Date | null;
}

// ==============================================================================
// Book & Ingestion (Phase 2 & 3)
// ==============================================================================

export interface HeadingItem {
  title: string;
  level: number;
  page_number: number;
  confidence: 'toc' | 'heuristic' | string;
  source: string;
}

export interface BookMetadata {
  book_id: string;
  title?: string;
  original_filename: string;
  safe_filename: string;
  file_hash: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  extraction_status: 'completed' | 'processing' | 'failed' | string;
  needs_ocr: boolean;
  ocr_warning: string | null;
  ocr_needed_pages: number[];
  empty_pages: number[];
  headings: HeadingItem[];
  created_at: string;
  indexing_status: 'not_indexed' | 'indexing' | 'indexed' | 'failed';
  indexed_chunk_count: number;
  indexing_error?: string | null;
  indexed_at?: string | null;
}

export interface ChunkMetadata {
  chunk_id: string;
  book_id: string;
  chunk_index: number;
  text: string;
  page_start: number;
  page_end: number;
  chapter: string | null;
  section: string | null;
  heading_confidence: string;
  char_count: number;
  word_count: number;
}

export interface BookDetailResponse {
  book: BookMetadata;
  page_reports: unknown[];
}

export interface BookListResponse {
  total: number;
  books: BookMetadata[];
}

// ==============================================================================
// Hierarchical Knowledge (Phase 4)
// ==============================================================================

export interface KeyConceptRecord {
  name: string;
  definition: string;
  claim_type: 'author_stated' | 'model_inferred';
  source_chunk_ids: string[];
  page_references: number[];
}

export interface DefinitionRecord {
  term: string;
  explanation: string;
  verbatim_or_paraphrased: 'verbatim' | 'paraphrased';
  source_chunk_id: string;
  page_number: number;
}

export interface SectionSummary {
  section_id: string;
  chapter_title: string;
  section_title: string;
  page_start: number;
  page_end: number;
  concise_summary: string;
  key_concepts: KeyConceptRecord[];
  definitions: DefinitionRecord[];
  reasoning_chain: string[];
  source_chunk_ids: string[];
  uncertainty?: string | null;
}

export interface ChapterSummary {
  chapter_index: number;
  chapter_title: string;
  page_start: number;
  page_end: number;
  chapter_crux: string;
  central_argument: string;
  supporting_arguments?: Array<{ claim: string; reasoning?: string; claim_type?: string; source_chunk_ids?: string[] }>;
  supporting_points?: Array<{ point?: string; claim?: string; claim_type?: string; source_chunk_ids?: string[] }>;
  important_assumptions?: Array<{ assumption: string; explicit_or_implicit?: string; source_chunk_ids?: string[] }>;
  examples?: Array<{ description: string; point_illustrated?: string; source_chunk_ids?: string[] }>;
  counterexamples?: Array<{ description: string; source_chunk_ids?: string[] }>;
  conclusions?: Array<{ statement: string; claim_type?: string; source_chunk_ids?: string[] }>;
  prerequisites_and_connections?: Array<{ related_chapter_title: string; nature_of_connection?: string }>;
  supporting_section_ids?: string[];
  source_chunk_ids?: string[];
  evidence_confidence?: 'high' | 'moderate' | 'insufficient' | string;
  uncertainty?: string | null;
}

export interface BookSummary {
  book_id: string;
  central_thesis: string;
  overall_crux: string;
  recurring_themes: Array<{ theme: string; manifestations: string; chapters_present: string[] }>;
  essential_takeaways: Array<{ point: string; explanation: string; claim_type: string; page_numbers: number[] }>;
  chapter_relationships: Array<{ source_chapter: string; target_chapter: string; relationship_type: string; description: string }>;
  important_caveats: string[];
  uncertainty?: string | null;
}

export interface BookHierarchy {
  book_id: string;
  generation_status: 'not_started' | 'in_progress' | 'completed' | 'failed';
  progress_percent: number;
  sections: SectionSummary[];
  chapters: ChapterSummary[];
  book_level: BookSummary | null;
  completed_units: string[];
  error_message?: string | null;
}

// ==============================================================================
// Conversational RAG (Phase 5)
// ==============================================================================

export type QuestionType =
  | 'factual'
  | 'conceptual'
  | 'chapter_summary'
  | 'book_crux'
  | 'cross_chapter_comparison'
  | 'quiz';

export interface Citation {
  chunk_id: string;
  page_start: number;
  page_end: number;
  chapter?: string | null;
  chapter_title?: string | null;
  section?: string | null;
  section_title?: string | null;
  quoted_snippet: string;
  confidence?: string | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: Citation[];
  question_type?: QuestionType;
  insufficient_evidence?: boolean;
  timestamp: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  top_k?: number;
  enable_reranker?: boolean;
  include_hierarchy?: boolean;
}

export interface ChatResponse {
  conversation_id: string;
  book_id: string;
  question_type: QuestionType;
  answer: string;
  crux_summary: string | null;
  citations: Citation[];
  insufficient_evidence: boolean;
  confidence: 'high' | 'moderate' | 'low' | 'insufficient';
  model_name: string;
  latency_ms: number;
}

export interface ConversationHistory {
  conversation_id: string;
  book_id: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export type Conversation = ConversationHistory;

// ==============================================================================
// Evaluation Framework (Phase 6)
// ==============================================================================

export type EvaluationQuestionType =
  | 'factual'
  | 'conceptual'
  | 'summarization'
  | 'multi_hop'
  | 'comparison'
  | 'unanswerable';

export type ValidationStatus = 'draft' | 'validated' | 'rejected';

export type PipelineType =
  | 'baseline_rag'
  | 'hybrid_rag'
  | 'hierarchical_rag'
  | 'hierarchical_rag_rerank';

export interface EvaluationItem {
  item_id: string;
  question: string;
  question_type: EvaluationQuestionType;
  reference_answer: string;
  expected_chunk_ids: string[];
  expected_page_numbers: number[];
  is_answerable: boolean;
  essential_concepts: string[];
  validation_status: ValidationStatus;
  notes?: string | null;
  created_at: string;
  updated_at: string;
}

export interface EvaluationDataset {
  dataset_id: string;
  book_id: string;
  version: string;
  description: string;
  items: EvaluationItem[];
  created_at: string;
  updated_at: string;
}

export interface RetrievalMetrics {
  recall_at_k: number;
  precision_at_k: number;
  mrr: number;
  hit_rate_at_k: number;
  k: number;
}

export interface CitationMetrics {
  citation_validity: number;
  citation_precision: number;
  citation_recall: number;
  total_citations: number;
  valid_citations: number;
}

export interface CruxRubricResult {
  essential_concepts_coverage: number;
  central_thesis_representation: number;
  important_qualifications_retained: number;
  supporting_reasoning_included: number;
  cross_chapter_connections: number;
  rubric_breakdown: Record<string, unknown>;
}

export interface GenerationMetrics {
  faithfulness: number;
  unsupported_claim_rate: number;
  answer_relevance: number;
  answer_correctness: number;
  context_relevance: number;
  abstention_quality: number;
  citations: CitationMetrics;
  crux_rubric: CruxRubricResult;
}

export interface SampleEvaluationResult {
  item_id: string;
  question: string;
  question_type: EvaluationQuestionType;
  pipeline_type: PipelineType;
  retrieved_chunk_ids: string[];
  generated_answer: string;
  crux_summary?: string | null;
  generated_citations: Citation[];
  insufficient_evidence: boolean;
  confidence: string;
  retrieval_metrics: RetrievalMetrics;
  generation_metrics: GenerationMetrics;
  latency_ms: number;
  is_failure: boolean;
  failure_reasons: string[];
}

export interface FailureCase {
  item_id: string;
  question: string;
  question_type: EvaluationQuestionType;
  pipeline_type: PipelineType;
  failure_reasons: string[];
  retrieved_chunk_ids: string[];
  expected_chunk_ids: string[];
  recall_at_k: number;
  faithfulness: number;
  unsupported_claim_rate: number;
  citation_validity: number;
  abstention_quality: number;
  notes?: string | null;
}

export interface AggregateMetrics {
  sample_count: number;
  mean_recall_at_k: number;
  mean_precision_at_k: number;
  mean_mrr: number;
  mean_hit_rate_at_k: number;
  mean_faithfulness: number;
  mean_unsupported_claim_rate: number;
  mean_answer_relevance: number;
  mean_answer_correctness: number;
  mean_context_relevance: number;
  mean_abstention_quality: number;
  mean_citation_validity: number;
  mean_citation_precision: number;
  mean_citation_recall: number;
  mean_essential_concepts_coverage: number;
  mean_central_thesis_representation: number;
  mean_important_qualifications_retained: number;
  mean_supporting_reasoning_included: number;
  mean_cross_chapter_connections: number;
  failure_case_count: number;
  mean_latency_ms: number;
}

export interface ExperimentRun {
  run_id: string;
  book_id: string;
  dataset_id: string;
  dataset_version: string;
  pipeline_type: PipelineType;
  pipeline_config: Record<string, unknown>;
  timestamp: string;
  created_at?: string;
  duration_seconds: number;
  sample_count?: number;
  top_k?: number;
  aggregate_metrics: AggregateMetrics;
  sample_results: SampleEvaluationResult[];
  failure_cases: FailureCase[];
}

export interface PipelineComparisonReport {
  book_id: string;
  dataset_id: string;
  dataset_version: string;
  timestamp: string;
  runs: Record<string, ExperimentRun>;
  comparative_summary: Array<Record<string, unknown>>;
  key_findings: string[];
}
