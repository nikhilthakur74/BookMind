import React from 'react';
import { UploadCloud, FileText, Lock } from 'lucide-react';

export const BookUploadPlaceholder: React.FC = () => {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-dashed border-[#DDD7CC] bg-[#F7F5EE]/70 p-8 text-center transition-all hover:border-[#C4BEB2]">
      <div className="mx-auto flex max-w-md flex-col items-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-[#DFD9CD] bg-[#FAF8F3] text-[#7A756C] shadow-xs">
          <UploadCloud className="h-6 w-6 text-[#A06044]" strokeWidth={1.75} />
        </div>

        <h3 className="mt-4 font-serif-claude text-lg font-medium text-[#22211F]">
          PDF Book Ingestion Dropzone
        </h3>

        <p className="mt-1.5 text-xs leading-relaxed text-[#78736A]">
          Drop your EPUB or PDF volumes here to extract structured chapters, section headings, and semantic chunks using PyMuPDF.
        </p>

        <div className="mt-5 flex items-center gap-2 rounded-full border border-[#E3DDD1] bg-[#F1ECE1] px-3.5 py-1 text-xs text-[#736E65]">
          <Lock className="h-3.5 w-3.5 text-[#A06044]" />
          <span className="font-medium">Unlocks in Phase 2 (PDF Processing Pipeline)</span>
        </div>

        <div className="mt-4 flex items-center gap-3 text-[11px] text-[#918C83]">
          <span className="flex items-center gap-1">
            <FileText className="h-3 w-3" /> PyMuPDF 1.24+
          </span>
          <span>•</span>
          <span>Target directory: <code className="font-mono-code text-[#4D4942]">data/uploads/</code></span>
        </div>
      </div>
    </div>
  );
};
