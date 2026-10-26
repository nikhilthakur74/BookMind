import { HealthCard } from '../components/HealthCard';
import { PipelineOverview } from '../components/PipelineOverview';
import { BookUploadPlaceholder } from '../components/BookUploadPlaceholder';
import type { HealthState } from '../types/api';
import { Sparkles, Terminal } from 'lucide-react';

interface HomeProps {
  healthState: HealthState;
  onRefreshHealth: () => void;
}

export function Home({ healthState, onRefreshHealth }: HomeProps) {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10 sm:px-8 space-y-12">
      {/* Hero Section with Claude-inspired typography and quiet elegance */}
      <section className="space-y-4 max-w-2xl">
        <div className="inline-flex items-center gap-1.5 rounded-full border border-[#E3DCCF] bg-[#F4EFE6] px-3 py-1 text-xs text-[#7A756C]">
          <Sparkles className="h-3 w-3 text-[#B05C38]" />
          <span className="font-medium text-[#4D4942]">Phase 1: Project Setup & Scaffolding</span>
        </div>

        <h1 className="font-serif-claude text-3xl sm:text-4xl lg:text-[42px] font-normal leading-tight tracking-tight text-[#22211F]">
          Understand books at depth, <br className="hidden sm:inline" />
          <span className="italic text-[#706B62]">not just in fragments.</span>
        </h1>

        <p className="text-sm leading-relaxed text-[#6E6960]">
          BookMind combines high-density document layout extraction, BGE-M3 semantic vectors, Qdrant vector indexing, and Groq ultra-fast inference for deep book comprehension and grounded thematic inquiry.
        </p>
      </section>

      {/* Backend Health Check Card */}
      <section>
        <HealthCard healthState={healthState} onRefresh={onRefreshHealth} />
      </section>

      {/* PDF Upload Dropzone (Phase 2 preview) */}
      <section>
        <BookUploadPlaceholder />
      </section>

      {/* Pipeline Architecture Stage Cards */}
      <section>
        <PipelineOverview />
      </section>

      {/* Quick Verification & Startup Reference Card */}
      <section className="rounded-2xl border border-[#ECE7DE] bg-[#FAF8F3] p-6 text-xs text-[#6E6960] space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[#22211F] font-medium">
            <Terminal className="h-4 w-4 text-[#8C867C]" />
            <span className="font-serif-claude text-base">Quick PowerShell Commands</span>
          </div>
          <span className="font-mono-code text-[11px] text-[#8C867C]">Windows Local Dev</span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3 space-y-1.5">
            <span className="font-medium text-[#22211F]">Backend Server (FastAPI):</span>
            <div className="rounded bg-[#24211E] px-2.5 py-1.5 text-[#EFEBE4]">
              <code className="font-mono-code text-[11px]">cd backend; .venv\Scripts\uvicorn app.main:app --reload</code>
            </div>
          </div>

          <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3 space-y-1.5">
            <span className="font-medium text-[#22211F]">Backend Health Test (pytest):</span>
            <div className="rounded bg-[#24211E] px-2.5 py-1.5 text-[#EFEBE4]">
              <code className="font-mono-code text-[11px]">cd backend; .venv\Scripts\pytest -v</code>
            </div>
          </div>
        </div>
      </section>

      {/* Neutral, understated footer */}
      <footer className="border-t border-[#ECE7DE] pt-8 pb-12 text-center text-xs text-[#8C867C]">
        <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2">
          <span>Python 3.12</span>
          <span>•</span>
          <span>FastAPI</span>
          <span>•</span>
          <span>PyMuPDF</span>
          <span>•</span>
          <span>BGE-M3</span>
          <span>•</span>
          <span>Qdrant</span>
          <span>•</span>
          <span>Groq API</span>
          <span>•</span>
          <span>React 19 + Vite</span>
          <span>•</span>
          <span>Tailwind CSS</span>
        </div>
        <p className="mt-3 text-[11px] text-[#A6A095]">
          BookMind RAG Engine — Clean Architecture & Modular Engineering
        </p>
      </footer>
    </div>
  );
}
