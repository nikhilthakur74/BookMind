import { useState, useEffect } from 'react';
import { X, Layers, Sparkles, BookOpen, ChevronDown, ChevronRight, RefreshCw, FileText } from 'lucide-react';
import { getHierarchy, generateHierarchy } from '../lib/api';
import type { BookHierarchy, BookMetadata } from '../types/api';

interface HierarchyModalProps {
  isOpen: boolean;
  book: BookMetadata | null;
  onClose: () => void;
}

export const HierarchyModal: React.FC<HierarchyModalProps> = ({ isOpen, book, onClose }) => {
  const [hierarchy, setHierarchy] = useState<BookHierarchy | null>(null);
  const [loading, setLoading] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [activeTab, setActiveTab] = useState<'book' | 'chapters' | 'sections'>('book');
  const [expandedChapter, setExpandedChapter] = useState<number | null>(0);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && book) {
      loadHierarchyData(book.book_id);
    }
  }, [isOpen, book]);

  const loadHierarchyData = async (bookId: string) => {
    setLoading(true);
    try {
      const data = await getHierarchy(bookId);
      setHierarchy(data);
    } catch {
      setHierarchy(null);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!book) return;
    setSynthesizing(true);
    try {
      const data = await generateHierarchy(book.book_id, true);
      setHierarchy(data);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Hierarchy generation failed.');
    } finally {
      setSynthesizing(false);
    }
  };

  if (!isOpen || !book) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 backdrop-blur-xs p-4">
      <div className="relative flex flex-col h-[85vh] w-full max-w-4xl rounded-2xl border border-[#E5E0D8] bg-[#FAF9F5] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#ECE7DE] bg-[#F6F3EC] px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-[#EAE4D8] p-2 text-[#C86432]">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <h2 className="font-serif-claude text-xl font-medium text-[#22211F]">
                Four Knowledge Levels: {book.original_filename}
              </h2>
              <p className="text-xs text-[#706B62]">
                Extract the crux without losing the evidence: Levels 2 to 4 synthesis.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleGenerate}
              disabled={synthesizing}
              className="inline-flex items-center gap-1.5 rounded-xl border border-[#D5CEC2] bg-[#FAF8F3] px-3 py-1.5 text-xs font-medium text-[#4D4942] hover:bg-[#EFEBE4] transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${synthesizing ? 'animate-spin text-[#C86432]' : ''}`} />
              {synthesizing ? 'Synthesizing...' : 'Regenerate'}
            </button>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-[#8C867C] hover:bg-[#ECE6DB] hover:text-[#22211F] transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-[#ECE7DE] bg-[#FAF8F3] px-6 gap-6 text-xs font-medium text-[#706B62]">
          <button
            onClick={() => setActiveTab('book')}
            className={`py-3 transition-colors border-b-2 flex items-center gap-1.5 ${
              activeTab === 'book'
                ? 'border-[#C86432] text-[#22211F] font-semibold'
                : 'border-transparent hover:text-[#3D3A36]'
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" /> Level 4: Book Crux & Thesis
          </button>
          <button
            onClick={() => setActiveTab('chapters')}
            className={`py-3 transition-colors border-b-2 flex items-center gap-1.5 ${
              activeTab === 'chapters'
                ? 'border-[#C86432] text-[#22211F] font-semibold'
                : 'border-transparent hover:text-[#3D3A36]'
            }`}
          >
            <BookOpen className="h-3.5 w-3.5" /> Level 3: Chapter Summaries ({hierarchy?.chapters.length || 0})
          </button>
          <button
            onClick={() => setActiveTab('sections')}
            className={`py-3 transition-colors border-b-2 flex items-center gap-1.5 ${
              activeTab === 'sections'
                ? 'border-[#C86432] text-[#22211F] font-semibold'
                : 'border-transparent hover:text-[#3D3A36]'
            }`}
          >
            <FileText className="h-3.5 w-3.5" /> Level 2: Section Key Concepts ({hierarchy?.sections.length || 0})
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-3">
              <RefreshCw className="h-7 w-7 animate-spin text-[#C86432]" />
              <p className="text-xs text-[#706B62]">Loading four-level hierarchical knowledge...</p>
            </div>
          ) : !hierarchy || hierarchy.generation_status === 'not_started' ? (
            <div className="flex flex-col items-center justify-center py-16 text-center space-y-4 max-w-md mx-auto">
              <div className="rounded-full bg-[#EAE4D8] p-4 text-[#706B62]">
                <Layers className="h-8 w-8 text-[#C86432]" />
              </div>
              <div className="space-y-1">
                <p className="font-serif-claude text-lg text-[#22211F]">Hierarchy Not Yet Generated</p>
                <p className="text-xs text-[#706B62]">
                  Synthesizing Level 2 section summaries, Level 3 chapter cruxes, and Level 4 overarching thesis enables deep thematic inquiries.
                </p>
              </div>
              <button
                onClick={handleGenerate}
                disabled={synthesizing}
                className="rounded-xl bg-[#C86432] px-4 py-2 text-xs font-medium text-white hover:bg-[#B35528] transition-colors"
              >
                {synthesizing ? 'Synthesizing...' : 'Generate Four Knowledge Levels'}
              </button>
            </div>
          ) : (
            <>
              {/* LEVEL 4: BOOK VIEW */}
              {activeTab === 'book' && hierarchy.book_level && (
                <div className="space-y-6 max-w-3xl">
                  {/* Central Thesis */}
                  <div className="rounded-2xl border border-[#E3DCcf] bg-[#FFFFFF] p-6 shadow-xs space-y-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[#C86432]">
                      Central Thesis
                    </span>
                    <p className="font-serif-claude text-xl text-[#22211F] leading-snug">
                      "{hierarchy.book_level.central_thesis}"
                    </p>
                  </div>

                  {/* Overall Crux */}
                  <div className="rounded-2xl border border-[#E3DCCF] bg-[#FAF8F3] p-6 space-y-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[#706B62]">
                      Overall Crux & Condensed Essence
                    </span>
                    <p className="text-sm leading-relaxed text-[#3D3A36]">
                      {hierarchy.book_level.overall_crux}
                    </p>
                  </div>

                  {/* Essential Takeaways */}
                  <div className="space-y-3">
                    <h3 className="font-serif-claude text-base font-medium text-[#22211F]">
                      Essential Grounded Takeaways
                    </h3>
                    <div className="grid grid-cols-1 gap-3">
                      {hierarchy.book_level.essential_takeaways.map((takeaway, idx) => (
                        <div key={idx} className="rounded-xl border border-[#ECE7DE] bg-white p-4 space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-medium text-[#22211F]">{takeaway.point}</span>
                            <span className="rounded bg-[#F2EFE8] px-2 py-0.5 font-mono-code text-[11px] text-[#706B62]">
                              {takeaway.claim_type === 'author_stated' ? 'Author-stated' : 'Model-inferred'}
                            </span>
                          </div>
                          <p className="text-xs text-[#6E6960] leading-relaxed">{takeaway.explanation}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recurring Themes */}
                  {hierarchy.book_level.recurring_themes.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="font-serif-claude text-base font-medium text-[#22211F]">
                        Recurring Motifs & Themes
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {hierarchy.book_level.recurring_themes.map((th, idx) => (
                          <div key={idx} className="rounded-xl border border-[#ECE7DE] bg-[#FAF8F3] p-4 space-y-1">
                            <span className="font-medium text-xs text-[#22211F]">{th.theme}</span>
                            <p className="text-xs text-[#706B62] leading-relaxed">{th.manifestations}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* LEVEL 3: CHAPTERS VIEW */}
              {activeTab === 'chapters' && (
                <div className="space-y-4 max-w-3xl">
                  {hierarchy.chapters.map((chap, cIdx) => {
                    const isExpanded = expandedChapter === cIdx;
                    return (
                      <div
                        key={cIdx}
                        className="rounded-xl border border-[#ECE7DE] bg-white overflow-hidden shadow-xs"
                      >
                        <div
                          onClick={() => setExpandedChapter(isExpanded ? null : cIdx)}
                          className="flex items-center justify-between p-4 cursor-pointer hover:bg-[#FAF8F3] transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#F0EDE6] font-mono-code text-xs text-[#706B62]">
                              {chap.chapter_index + 1}
                            </span>
                            <div>
                              <p className="font-serif-claude text-base font-medium text-[#22211F]">
                                {chap.chapter_title}
                              </p>
                              <p className="text-[11px] text-[#8C867C]">
                                Pages {chap.page_start}–{chap.page_end} • {
                                  (chap.supporting_arguments || chap.supporting_points || []).length
                                } points
                              </p>
                            </div>
                          </div>
                          {isExpanded ? <ChevronDown className="h-4 w-4 text-[#8C867C]" /> : <ChevronRight className="h-4 w-4 text-[#8C867C]" />}
                        </div>

                        {isExpanded && (
                          <div className="border-t border-[#ECE7DE] bg-[#FAF9F5] p-5 space-y-4 text-xs">
                            <div className="space-y-1">
                              <span className="font-semibold text-[#C86432] uppercase tracking-wider text-[10px]">
                                Chapter Crux
                              </span>
                              <p className="text-sm font-serif-claude text-[#22211F]">{chap.chapter_crux}</p>
                            </div>
                            <div className="space-y-1">
                              <span className="font-medium text-[#706B62]">Central Argument:</span>
                              <p className="text-[#3D3A36] leading-relaxed">{chap.central_argument}</p>
                            </div>

                            {/* Supporting Points */}
                            <div className="space-y-1.5 pt-1">
                              <span className="font-medium text-[#706B62]">Supporting Premises:</span>
                              <ul className="list-disc pl-4 space-y-1 text-[#4D4942]">
                                {(chap.supporting_arguments || chap.supporting_points || []).map((p: any, pIdx: number) => (
                                  <li key={pIdx}>{p.claim || p.point || (typeof p === 'string' ? p : JSON.stringify(p))}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* LEVEL 2: SECTIONS VIEW */}
              {activeTab === 'sections' && (
                <div className="space-y-4 max-w-3xl">
                  {hierarchy.sections.map((sec) => {
                    const isExpanded = expandedSection === sec.section_id;
                    return (
                      <div key={sec.section_id} className="rounded-xl border border-[#ECE7DE] bg-white p-4 space-y-3">
                        <div
                          onClick={() => setExpandedSection(isExpanded ? null : sec.section_id)}
                          className="flex items-center justify-between cursor-pointer"
                        >
                          <div>
                            <p className="font-medium text-xs text-[#22211F]">
                              {sec.chapter_title} — {sec.section_title}
                            </p>
                            <p className="text-[11px] text-[#8C867C]">
                              Pages {sec.page_start}–{sec.page_end} • {
                                (Array.isArray(sec.key_concepts) ? sec.key_concepts : []).length
                              } key concepts
                            </p>
                          </div>
                          {isExpanded ? <ChevronDown className="h-4 w-4 text-[#8C867C]" /> : <ChevronRight className="h-4 w-4 text-[#8C867C]" />}
                        </div>

                        <p className="text-xs text-[#59554E] leading-relaxed">{sec.concise_summary}</p>

                        {isExpanded && Array.isArray(sec.key_concepts) && sec.key_concepts.length > 0 && (
                          <div className="border-t border-[#ECE7DE] pt-3 space-y-2">
                            <span className="text-[11px] font-medium text-[#706B62]">Concepts & Definitions:</span>
                            {sec.key_concepts.map((kc, kIdx) => (
                              <div key={kIdx} className="rounded-lg bg-[#FAF8F3] p-2.5 text-xs space-y-0.5 border border-[#ECE7DE]">
                                <span className="font-semibold text-[#22211F]">{kc.name}</span>
                                <p className="text-[#6E6960]">{kc.definition}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
