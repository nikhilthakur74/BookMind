import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  BookOpen,
  Layers,
  AlertTriangle,
  RotateCcw,
  SlidersHorizontal,
  Check,
  Copy,
  Lightbulb,
  Compass,
} from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { CitationsView } from './CitationsView';
import type {
  BookMetadata,
  ChatMessage,
  PipelineType,
  BookHierarchy,
} from '../types/api';

interface ChatWorkspaceProps {
  book: BookMetadata | null;
  hierarchy: BookHierarchy | null;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onSendMessage: (
    message: string,
    options?: {
      pipeline?: PipelineType;
      enableRerank?: boolean;
      topK?: number;
    }
  ) => void;
  onRetryLastMessage?: () => void;
  onOpenHierarchy: () => void;
  onOpenUpload: () => void;
}

const SUGGESTED_QUESTIONS = [
  {
    type: 'Crux',
    text: 'What is the central thesis and overall crux of this book?',
    category: 'book_crux',
  },
  {
    type: 'Conceptual',
    text: 'Explain the core mechanisms and principles governing the author’s primary argument.',
    category: 'conceptual',
  },
  {
    type: 'Chapter Summary',
    text: 'Provide a structured summary of the arguments and takeaways in Chapter 1.',
    category: 'chapter_summary',
  },
  {
    type: 'Comparison',
    text: 'Compare and contrast the contrasting models or approaches presented across chapters.',
    category: 'cross_chapter_comparison',
  },
  {
    type: 'Factual',
    text: 'What specific conditions or criteria must be satisfied according to the author?',
    category: 'factual',
  },
];

export function ChatWorkspace({
  book,
  hierarchy,
  messages,
  isLoading,
  error,
  onSendMessage,
  onRetryLastMessage,
  onOpenHierarchy,
  onOpenUpload,
}: ChatWorkspaceProps) {
  const [inputText, setInputText] = useState('');
  const [pipeline, setPipeline] = useState<PipelineType>('hierarchical_rag');
  const [enableRerank, setEnableRerank] = useState(false);
  const [topK, setTopK] = useState(5);
  const [showOptions, setShowOptions] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle textarea auto-resize
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [inputText]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isLoading || !book) return;

    onSendMessage(trimmed, {
      pipeline,
      enableRerank,
      topK,
    });
    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  if (!book) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-[#FAF9F5]">
        <div className="max-w-md space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-[#F0EBE0] flex items-center justify-center mx-auto text-[#C86432]">
            <BookOpen className="w-6 h-6" />
          </div>
          <h2 className="font-serif-claude text-2xl font-normal text-[#22211F]">
            No Book Selected
          </h2>
          <p className="text-sm text-[#736E65] leading-relaxed">
            Select a book from your library or upload a PDF to extract hierarchical knowledge, explore the central crux, and start asking grounded questions.
          </p>
          <button
            onClick={onOpenUpload}
            className="inline-flex items-center gap-2 rounded-xl bg-[#24211E] px-4 py-2.5 text-xs font-medium text-[#FAF9F5] hover:bg-[#383430] transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5 text-[#E6A07E]" />
            Upload Book PDF
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-[#FAF9F5] overflow-hidden">
      {/* Book Context Banner */}
      <div className="border-b border-[#ECE7DE] bg-[#FAF8F3] px-6 py-2.5 flex items-center justify-between text-xs">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2 font-medium text-[#22211F] truncate">
            <BookOpen className="w-4 h-4 text-[#C86432] flex-shrink-0" />
            <span className="truncate">{book.title || book.original_filename}</span>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-[#8C867C] font-mono-code text-[11px]">
            <span>{book.page_count} pp</span>
            <span>•</span>
            <span>{book.chunk_count || 0} chunks</span>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={onOpenHierarchy}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#DDD6C9] bg-[#FAF9F5] px-2.5 py-1 text-xs font-medium text-[#4D4942] hover:bg-[#F2EDE2] transition-colors"
          >
            <Layers className="w-3.5 h-3.5 text-[#B05C38]" />
            <span>Crux & Levels</span>
          </button>

          <button
            onClick={() => setShowOptions(!showOptions)}
            className={`p-1.5 rounded-lg border transition-colors ${
              showOptions
                ? 'border-[#C86432] bg-[#F7EFE8] text-[#C86432]'
                : 'border-[#DDD6C9] bg-[#FAF9F5] text-[#6E6960] hover:bg-[#F2EDE2]'
            }`}
            title="Retrieval & Pipeline options"
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Advanced Options Bar (collapsible) */}
      {showOptions && (
        <div className="border-b border-[#ECE7DE] bg-[#F4EFE6]/70 px-6 py-3 text-xs flex flex-wrap items-center gap-6 text-[#524D44] animate-fadeIn">
          <div className="flex items-center gap-2">
            <span className="font-medium text-[#22211F]">Pipeline:</span>
            <select
              value={pipeline}
              onChange={(e) => setPipeline(e.target.value as PipelineType)}
              className="rounded-md border border-[#D5CEC2] bg-[#FAF9F5] px-2 py-1 text-xs text-[#2D2B28] focus:border-[#C86432] focus:outline-none"
            >
              <option value="hierarchical_rag">Hierarchical RAG (Levels 1-4)</option>
              <option value="hybrid_rag">Hybrid RAG (Vector + BM25)</option>
              <option value="baseline_rag">Baseline RAG (Vector Only)</option>
            </select>
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enableRerank}
              onChange={(e) => setEnableRerank(e.target.checked)}
              className="rounded border-[#D5CEC2] text-[#C86432] focus:ring-[#C86432]"
            />
            <span>Enable Reranker</span>
          </label>

          <div className="flex items-center gap-2">
            <span>Evidence top_k:</span>
            <input
              type="range"
              min="1"
              max="10"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-20 accent-[#C86432]"
            />
            <span className="font-mono-code font-medium text-[#22211F]">{topK}</span>
          </div>
        </div>
      )}

      {/* Chat Messages Scroll Container */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-8">
          {/* Empty state when conversation has no messages */}
          {messages.length === 0 ? (
            <div className="space-y-8 py-6">
              {/* Claude-inspired calm greeting */}
              <div className="space-y-3">
                <div className="inline-flex items-center gap-1.5 text-xs font-medium text-[#B05C38] bg-[#F7EFE8] px-2.5 py-1 rounded-full border border-[#E8DCCF]">
                  <Compass className="w-3.5 h-3.5" />
                  <span>Deep Book Comprehension</span>
                </div>
                <h1 className="font-serif-claude text-2xl sm:text-3xl text-[#22211F] font-normal leading-snug">
                  What would you like to explore in <br className="hidden sm:inline" />
                  <span className="italic text-[#706B62]">{book.title || book.original_filename}</span>?
                </h1>
                {hierarchy?.book_level?.overall_crux && (
                  <p className="text-xs sm:text-sm text-[#666056] leading-relaxed border-l-2 border-[#C86432] pl-3 py-1 bg-[#F9F6F0] rounded-r-md">
                    <strong className="font-medium text-[#22211F]">Crux: </strong>
                    {hierarchy.book_level.overall_crux}
                  </p>
                )}
              </div>

              {/* Suggested Questions Grid */}
              <div className="space-y-2.5">
                <div className="flex items-center gap-1.5 text-xs text-[#8C867C] font-medium">
                  <Lightbulb className="w-3.5 h-3.5 text-[#B05C38]" />
                  <span>Suggested inquiries across knowledge levels</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {SUGGESTED_QUESTIONS.map((sq, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => onSendMessage(sq.text, { pipeline, enableRerank, topK })}
                      className="text-left rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-3 text-xs hover:border-[#D5CEC2] hover:bg-[#F5EFE4] transition-all group"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-[#9E988D] group-hover:text-[#B05C38] transition-colors">
                          {sq.type}
                        </span>
                        <Send className="w-2.5 h-2.5 text-[#B05C38] opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                      <p className="text-[#3F3B35] leading-relaxed font-sans">{sq.text}</p>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isUser = msg.role === 'user';

              return (
                <div
                  key={idx}
                  className={`flex flex-col space-y-2 ${
                    isUser ? 'items-end' : 'items-start'
                  }`}
                >
                  {/* Message Bubble Container */}
                  <div
                    className={`max-w-2xl text-sm leading-relaxed ${
                      isUser
                        ? 'rounded-2xl rounded-tr-xs bg-[#EFEAE1] px-4 py-3 text-[#22211F] font-sans'
                        : 'w-full rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-5 sm:p-6 shadow-xs text-[#2D2B28]'
                    }`}
                  >
                    {/* Assistant Message Header Metadata */}
                    {!isUser && (
                      <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#ECE7DE] text-xs">
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 rounded-full bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
                            <BookOpen className="w-3 h-3 text-[#E6A07E]" />
                          </div>
                          <span className="font-serif-claude font-medium text-[#22211F]">
                            BookMind
                          </span>
                          {msg.question_type && (
                            <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#F2EDE2] text-[#696357]">
                              {msg.question_type.replace(/_/g, ' ')}
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-1.5 text-[#8C867C]">
                          <button
                            onClick={() => handleCopy(msg.content, idx)}
                            title="Copy response"
                            className="p-1 rounded hover:bg-[#EFEAE1] transition-colors text-[#8C867C] hover:text-[#22211F]"
                          >
                            {copiedIndex === idx ? (
                              <Check className="w-3.5 h-3.5 text-emerald-600" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Insufficient Evidence Warning Banner */}
                    {!isUser && msg.insufficient_evidence && (
                      <div className="mb-3 rounded-lg border border-amber-200/80 bg-amber-50/70 p-2.5 text-xs text-amber-800 flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <strong className="font-medium">Insufficient Direct Grounding: </strong>
                          The book text does not contain explicit verified proof for this query. The answer admits uncertainty rather than hallucinating.
                        </div>
                      </div>
                    )}

                    {/* Message Body Content */}
                    {isUser ? (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    ) : (
                      <MarkdownRenderer content={msg.content} />
                    )}

                    {/* Expandable Verified Citations */}
                    {!isUser && msg.citations && msg.citations.length > 0 && (
                      <CitationsView citations={msg.citations} />
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Assistant Loading State */}
          {isLoading && (
            <div className="flex items-start max-w-2xl w-full">
              <div className="w-full rounded-2xl border border-[#ECE7DE] bg-[#FAF9F5] p-5 shadow-xs space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full bg-[#272522] flex items-center justify-center text-[#FAF9F5]">
                    <Sparkles className="w-3 h-3 text-[#E6A07E] animate-pulse" />
                  </div>
                  <span className="font-serif-claude font-medium text-xs text-[#22211F]">
                    Synthesizing grounded answer...
                  </span>
                </div>
                <div className="space-y-2 py-1">
                  <div className="h-3 bg-[#EFEAE1] rounded w-5/6 animate-pulse" />
                  <div className="h-3 bg-[#EFEAE1] rounded w-4/6 animate-pulse" />
                  <div className="h-3 bg-[#EFEAE1] rounded w-3/6 animate-pulse" />
                </div>
              </div>
            </div>
          )}

          {/* Error Banner with Retry State */}
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50/80 p-4 text-xs text-rose-800 flex items-start justify-between gap-3 shadow-xs">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="font-medium block text-rose-900">Request Error</strong>
                  <p className="mt-0.5">{error}</p>
                </div>
              </div>
              {onRetryLastMessage && (
                <button
                  onClick={onRetryLastMessage}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-rose-300 bg-white px-2.5 py-1 font-medium text-rose-700 hover:bg-rose-50 transition-colors flex-shrink-0"
                >
                  <RotateCcw className="w-3 h-3" />
                  Retry
                </button>
              )}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat Input Bar */}
      <div className="border-t border-[#ECE7DE] bg-[#FAF9F5] px-4 sm:px-8 py-3.5">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="relative">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask anything about ${book.title || book.original_filename}...`}
              disabled={isLoading}
              className="w-full rounded-2xl border border-[#D9D1C3] bg-[#FAF8F3] px-4 py-3 pr-14 text-sm text-[#2D2B28] placeholder-[#9E988D] focus:border-[#C86432] focus:bg-white focus:outline-none transition-all shadow-xs resize-none"
              style={{ minHeight: '48px', maxHeight: '180px' }}
            />

            <button
              type="submit"
              disabled={!inputText.trim() || isLoading}
              className="absolute right-2.5 bottom-2.5 p-2 rounded-xl bg-[#24211E] text-[#FAF9F5] hover:bg-[#383430] disabled:bg-[#DDD8CF] disabled:text-[#A8A296] disabled:cursor-not-allowed transition-all shadow-xs active:scale-95"
              title="Send message (Enter)"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>

          <div className="mt-2 flex items-center justify-between text-[11px] text-[#8C867C] px-1 font-sans">
            <span>
              Press <kbd className="rounded border border-[#D5CEC2] bg-[#EFEAE1] px-1 font-mono-code text-[10px]">Enter</kbd> to submit, <kbd className="rounded border border-[#D5CEC2] bg-[#EFEAE1] px-1 font-mono-code text-[10px]">Shift+Enter</kbd> for newline
            </span>
            <span className="hidden sm:inline font-mono-code">
              Pipeline: {pipeline.replace(/_/g, ' ')}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
