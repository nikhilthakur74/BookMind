import React from 'react';
import { Layers, FileText, Database, Sparkles, CheckCircle2 } from 'lucide-react';

interface Stage {
  number: string;
  title: string;
  tool: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'active' | 'upcoming';
}

const STAGES: Stage[] = [
  {
    number: '01',
    title: 'Project Scaffolding',
    tool: 'FastAPI + Vite + TS',
    description: 'Clean backend architecture, Pydantic settings, CORS, health test suite, and minimal UI.',
    icon: Layers,
    status: 'active',
  },
  {
    number: '02',
    title: 'PDF Processing',
    tool: 'PyMuPDF (fitz)',
    description: 'Extract multi-page text, preserve document hierarchy, inspect layout, and prepare clean chunk streams.',
    icon: FileText,
    status: 'upcoming',
  },
  {
    number: '03',
    title: 'Dense Vectors',
    tool: 'BGE-M3 + Qdrant',
    description: 'Multi-lingual dense representation stored and indexed in high-performance Qdrant vector database.',
    icon: Database,
    status: 'upcoming',
  },
  {
    number: '04',
    title: 'Synthesis & RAG',
    tool: 'Groq (Llama 3.3)',
    description: 'Ultra-low latency reasoning, grounded citations, and full-context synthesis across long book passages.',
    icon: Sparkles,
    status: 'upcoming',
  },
];

export const PipelineOverview: React.FC = () => {
  return (
    <section className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <h2 className="font-serif-claude text-xl font-medium tracking-tight text-[#22211F]">
            Incremental Pipeline Architecture
          </h2>
          <p className="text-xs text-[#7A756C]">
            Structured execution plan following modular senior engineering principles.
          </p>
        </div>
        <span className="font-mono-code text-xs text-[#8C867C]">1 / 4 Stages Active</span>
      </div>

      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2">
        {STAGES.map((stage) => {
          const Icon = stage.icon;
          const isActive = stage.status === 'active';

          return (
            <div
              key={stage.number}
              className={`group relative rounded-2xl border p-5 transition-all ${
                isActive
                  ? 'border-[#DFD7CB] bg-[#FAF8F3] shadow-xs'
                  : 'border-[#EBE7DF] bg-[#FAF9F6]/60 opacity-80 hover:opacity-100 hover:border-[#DFD7CB]'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-lg border ${
                      isActive
                        ? 'border-[#DFD7CB] bg-[#F2EDE2] text-[#22211F]'
                        : 'border-[#ECE7DE] bg-[#F5F2EA] text-[#8C867C]'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono-code text-[11px] text-[#8C867C]">
                        {stage.number}
                      </span>
                      <h4 className="font-serif-claude text-base font-medium text-[#22211F]">
                        {stage.title}
                      </h4>
                    </div>
                    <span className="font-mono-code text-[11px] text-[#A06044]">
                      {stage.tool}
                    </span>
                  </div>
                </div>

                {isActive ? (
                  <span className="inline-flex items-center gap-1 rounded-full border border-[#CFE2CE] bg-[#EDF7EC] px-2 py-0.5 text-[10px] font-medium text-[#2F6830]">
                    <CheckCircle2 className="h-3 w-3" />
                    <span>Implemented</span>
                  </span>
                ) : (
                  <span className="rounded-full border border-[#E5E0D5] bg-[#F2EEE4] px-2 py-0.5 text-[10px] font-medium text-[#7A756C]">
                    Next Phase
                  </span>
                )}
              </div>

              <p className="mt-3.5 text-xs leading-relaxed text-[#6E6960]">
                {stage.description}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
};
