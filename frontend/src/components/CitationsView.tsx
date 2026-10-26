import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  BookOpen,
  Quote,
  Layers,
  FileCheck,
} from 'lucide-react';
import type { Citation } from '../types/api';

interface CitationsViewProps {
  citations: Citation[];
}

export function CitationsView({ citations }: CitationsViewProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [allExpanded, setAllExpanded] = useState(false);

  if (!citations || citations.length === 0) {
    return null;
  }

  const toggleExpand = (idx: number) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  const toggleExpandAll = () => {
    setAllExpanded(!allExpanded);
    setExpandedIndex(null);
  };

  return (
    <div className="mt-4 pt-3 border-t border-[#ECE7DE] text-xs space-y-2">
      {/* Header bar */}
      <div className="flex items-center justify-between text-[#736E65]">
        <div className="flex items-center gap-1.5 font-medium">
          <FileCheck className="w-3.5 h-3.5 text-[#B05C38]" />
          <span>
            {citations.length} Verified {citations.length === 1 ? 'Source Passage' : 'Source Passages'}
          </span>
        </div>
        <button
          onClick={toggleExpandAll}
          className="text-[11px] text-[#A09A8F] hover:text-[#4A4741] transition-colors"
        >
          {allExpanded ? 'Collapse All' : 'Expand All'}
        </button>
      </div>

      {/* Citation item badges & accordions */}
      <div className="space-y-1.5">
        {citations.map((cite, idx) => {
          const isItemExpanded = allExpanded || expandedIndex === idx;
          const pageLabel =
            cite.page_start === cite.page_end
              ? `Page ${cite.page_start}`
              : `Pages ${cite.page_start}–${cite.page_end}`;

          return (
            <div
              key={`${cite.chunk_id || 'cite'}_${idx}`}
              className={`rounded-lg border transition-all ${
                isItemExpanded
                  ? 'border-[#D9D1C3] bg-[#FAF8F3] shadow-xs'
                  : 'border-[#EDE8DF] bg-[#FAF9F5] hover:border-[#DED6C8]'
              }`}
            >
              {/* Collapsed view header */}
              <button
                type="button"
                onClick={() => toggleExpand(idx)}
                className="w-full text-left px-3 py-2 flex items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#F2EDE2] text-[#47433B] text-[11px] font-medium font-sans">
                    <BookOpen className="w-2.5 h-2.5 text-[#B05C38]" />
                    {pageLabel}
                  </span>

                  {cite.chunk_id && (
                    <span className="font-mono-code text-[10px] text-[#8C867C] truncate max-w-[150px]">
                      {cite.chunk_id}
                    </span>
                  )}

                  {(cite.chapter_title || cite.chapter) && (
                    <span className="hidden sm:inline text-[11px] text-[#706B62] truncate max-w-[200px]">
                      • {cite.chapter_title || cite.chapter}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-1.5 flex-shrink-0 text-[#8C867C]">
                  {cite.confidence && (
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-[#9E988D]">
                      {cite.confidence}
                    </span>
                  )}
                  {isItemExpanded ? (
                    <ChevronDown className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5" />
                  )}
                </div>
              </button>

              {/* Expanded details view */}
              {isItemExpanded && (
                <div className="px-3 pb-3 pt-1 border-t border-[#F0EBE1] space-y-2">
                  {/* Chapter / Section breadcrumb */}
                  {(cite.chapter_title || cite.chapter || cite.section_title || cite.section) && (
                    <div className="flex items-center gap-1.5 text-[11px] text-[#706B62]">
                      <Layers className="w-3 h-3 text-[#A8A296]" />
                      <span>{cite.chapter_title || cite.chapter || 'Chapter'}</span>
                      {(cite.section_title || cite.section) && (
                        <>
                          <span>›</span>
                          <span className="font-medium text-[#4D4942]">
                            {cite.section_title || cite.section}
                          </span>
                        </>
                      )}
                    </div>
                  )}

                  {/* Verbatim quote snippet */}
                  <div className="relative pl-3 border-l-2 border-[#C86432] bg-[#F5F1E8]/70 rounded-r-md py-2 pr-2.5 text-xs text-[#3E3A34] leading-relaxed italic">
                    <Quote className="w-3 h-3 text-[#C86432]/60 absolute -top-1.5 -left-1.5 bg-[#FAF8F3] rounded-full" />
                    "{cite.quoted_snippet}"
                  </div>

                  {/* Chunk Metadata details */}
                  <div className="flex flex-wrap items-center justify-between text-[10px] text-[#8C867C] font-mono-code pt-1">
                    <span>Chunk: {cite.chunk_id}</span>
                    <span>Exact Pages: {cite.page_start}–{cite.page_end}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
