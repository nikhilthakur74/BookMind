import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Loader2, X, FileText } from 'lucide-react';
import { uploadBook, indexBook, generateHierarchy } from '../lib/api';
import type { BookMetadata } from '../types/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (book: BookMetadata) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUploadSuccess }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [autoGenerateHierarchy, setAutoGenerateHierarchy] = useState(true);
  const [step, setStep] = useState<'idle' | 'uploading' | 'indexing' | 'hierarchy' | 'done' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndSetFile = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('Only PDF files are supported.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 50 MB limit.');
      return;
    }
    setErrorMessage(null);
    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleStartIngestion = async () => {
    if (!selectedFile) return;

    try {
      // Step 1: Uploading & Extraction
      setStep('uploading');
      setStatusMessage('Extracting text, layout headings, and OCR evaluation...');
      const uploadRes = await uploadBook(selectedFile);
      const newBook = uploadRes.book;

      // Step 2: Indexing Vectors & BM25
      setStep('indexing');
      setStatusMessage('Generating BGE-M3 embeddings and indexing in Qdrant & BM25...');
      const indexedBook = await indexBook(newBook.book_id, true);

      // Step 3: Optional Level 2-4 Hierarchies
      if (autoGenerateHierarchy) {
        setStep('hierarchy');
        setStatusMessage('Synthesizing section cruxes, chapter summaries, and book thesis...');
        try {
          await generateHierarchy(newBook.book_id, false);
        } catch {
          // If hierarchy fails or times out, proceed anyway so the book is usable
        }
      }

      setStep('done');
      setStatusMessage('Book successfully ingested and ready for grounded exploration.');
      setTimeout(() => {
        onUploadSuccess(indexedBook);
        onClose();
      }, 1200);
    } catch (err: unknown) {
      setStep('error');
      setErrorMessage(err instanceof Error ? err.message : 'An error occurred during ingestion.');
    }
  };

  const resetModal = () => {
    setSelectedFile(null);
    setStep('idle');
    setErrorMessage(null);
    setStatusMessage('');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 backdrop-blur-xs p-4">
      <div className="relative w-full max-w-lg rounded-2xl border border-[#E5E0D8] bg-[#FAF9F5] p-6 shadow-xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#ECE7DE] pb-4">
          <div className="space-y-0.5">
            <h2 className="font-serif-claude text-xl font-medium text-[#22211F]">Ingest a New Book</h2>
            <p className="text-xs text-[#706B62]">Upload a PDF for multi-level extraction and grounded RAG.</p>
          </div>
          <button
            onClick={() => {
              resetModal();
              onClose();
            }}
            disabled={step === 'uploading' || step === 'indexing'}
            className="rounded-lg p-1.5 text-[#8C867C] hover:bg-[#F0EDE6] hover:text-[#22211F] transition-colors disabled:opacity-50"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="flex items-start gap-2.5 rounded-xl border border-[#E8C4B8] bg-[#FDF4F0] p-3 text-xs text-[#9E3E24]">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-medium">Ingestion Issue:</span> {errorMessage}
            </div>
          </div>
        )}

        {/* Content based on Step */}
        {step === 'idle' && (
          <div className="space-y-4">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-[#C86432] bg-[#FAF5EE]'
                  : selectedFile
                  ? 'border-[#A39D92] bg-[#F4EFE6]'
                  : 'border-[#DDD8CE] bg-[#FAF8F3] hover:border-[#B5AFA4] hover:bg-[#F5F2EB]'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={handleFileChange}
              />
              {selectedFile ? (
                <div className="flex flex-col items-center space-y-2">
                  <div className="rounded-full bg-[#E8E2D5] p-3 text-[#22211F]">
                    <FileText className="h-6 w-6 text-[#C86432]" />
                  </div>
                  <div className="space-y-0.5">
                    <p className="font-medium text-sm text-[#22211F] max-w-xs truncate">{selectedFile.name}</p>
                    <p className="text-xs text-[#706B62]">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  <span className="text-[11px] text-[#A6A095] underline pt-1">Click to change file</span>
                </div>
              ) : (
                <div className="flex flex-col items-center space-y-2">
                  <div className="rounded-full bg-[#F0EDE6] p-3 text-[#706B62]">
                    <UploadCloud className="h-6 w-6" />
                  </div>
                  <p className="text-sm font-medium text-[#22211F]">Drag and drop your PDF here</p>
                  <p className="text-xs text-[#8C867C]">or click to browse from your device</p>
                  <span className="text-[11px] text-[#A6A095] pt-1">Maximum file size: 50 MB</span>
                </div>
              )}
            </div>

            {/* Options */}
            <div className="flex items-center gap-2 pt-1">
              <input
                type="checkbox"
                id="autoGenHier"
                checked={autoGenerateHierarchy}
                onChange={(e) => setAutoGenerateHierarchy(e.target.checked)}
                className="h-4 w-4 rounded border-[#D0C9BD] text-[#C86432] focus:ring-0 focus:ring-offset-0"
              />
              <label htmlFor="autoGenHier" className="text-xs text-[#4D4942] cursor-pointer">
                Auto-generate Level 2–4 Hierarchical Knowledge after indexing
              </label>
            </div>
          </div>
        )}

        {/* Loading / Progress State */}
        {(step === 'uploading' || step === 'indexing' || step === 'hierarchy') && (
          <div className="flex flex-col items-center justify-center py-8 space-y-4 text-center">
            <Loader2 className="h-8 w-8 animate-spin text-[#C86432]" />
            <div className="space-y-1">
              <p className="font-medium text-sm text-[#22211F]">
                {step === 'uploading' && 'Extracting PDF & Chunks'}
                {step === 'indexing' && 'Indexing Vectors & BM25'}
                {step === 'hierarchy' && 'Synthesizing Four Knowledge Levels'}
              </p>
              <p className="text-xs text-[#706B62] max-w-sm">{statusMessage}</p>
            </div>
          </div>
        )}

        {/* Done State */}
        {step === 'done' && (
          <div className="flex flex-col items-center justify-center py-6 space-y-3 text-center">
            <CheckCircle2 className="h-10 w-10 text-[#3B7A57]" />
            <p className="font-medium text-sm text-[#22211F]">{statusMessage}</p>
          </div>
        )}

        {/* Footer Actions */}
        <div className="flex items-center justify-end gap-3 border-t border-[#ECE7DE] pt-4">
          <button
            type="button"
            onClick={() => {
              resetModal();
              onClose();
            }}
            disabled={step === 'uploading' || step === 'indexing'}
            className="rounded-xl border border-[#DCD6CA] px-4 py-2 text-xs font-medium text-[#59554E] hover:bg-[#F0EDE6] transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          {step === 'idle' && (
            <button
              type="button"
              disabled={!selectedFile}
              onClick={handleStartIngestion}
              className="rounded-xl bg-[#C86432] px-4 py-2 text-xs font-medium text-white hover:bg-[#B35528] transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-xs"
            >
              Start Ingestion
            </button>
          )}
          {step === 'error' && (
            <button
              type="button"
              onClick={resetModal}
              className="rounded-xl bg-[#22211F] px-4 py-2 text-xs font-medium text-white hover:bg-[#383531] transition-colors"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
