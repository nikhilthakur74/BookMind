import { useState } from 'react';
import { Activity, CheckCircle2, AlertTriangle, Terminal, Code2, Clock, Globe, Cpu } from 'lucide-react';
import type { HealthState } from '../types/api';

interface HealthCardProps {
  healthState: HealthState;
  onRefresh: () => void;
}

export function HealthCard({ healthState, onRefresh }: HealthCardProps) {
  const [showJson, setShowJson] = useState(false);
  const isHealthy = healthState.data?.status === 'healthy';

  return (
    <div className="overflow-hidden rounded-2xl border border-[#ECE7DE] bg-[#FAF8F3] p-6 shadow-xs transition-all">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-xl border ${
              isHealthy
                ? 'border-[#CFE2CE] bg-[#EDF7EC] text-[#2F6830]'
                : 'border-[#ECD0CC] bg-[#FDF1F0] text-[#9A3428]'
            }`}
          >
            {isHealthy ? (
              <CheckCircle2 className="h-5 w-5" strokeWidth={2} />
            ) : (
              <AlertTriangle className="h-5 w-5" strokeWidth={2} />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-serif-claude text-lg font-medium text-[#22211F]">
                Backend Health Check
              </h3>
              <span className="font-mono-code text-[11px] text-[#7A756C]">
                GET /api/health
              </span>
            </div>
            <p className="text-xs text-[#7A756C]">
              Verified contract response and CORS readiness for local development.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowJson(!showJson)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2DDCF] bg-[#FAF9F5] px-3 py-1.5 text-xs font-medium text-[#4D4942] transition-colors hover:bg-[#F3EFE6]"
          >
            <Code2 className="h-3.5 w-3.5 text-[#7A756C]" />
            <span>{showJson ? 'Hide JSON' : 'Inspect JSON'}</span>
          </button>
          <button
            onClick={onRefresh}
            disabled={healthState.loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#373531] bg-[#2A2825] px-3.5 py-1.5 text-xs font-medium text-[#FAF9F5] transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            <Activity className={`h-3.5 w-3.5 text-[#E6A07E] ${healthState.loading ? 'animate-spin' : ''}`} />
            <span>{healthState.loading ? 'Testing...' : 'Test Connection'}</span>
          </button>
        </div>
      </div>

      {/* Status metrics grid */}
      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3.5">
          <span className="text-[11px] font-medium tracking-wide text-[#8C867C] uppercase">Status</span>
          <div className="mt-1 flex items-center gap-1.5">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                isHealthy ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <span className="font-mono-code text-sm font-medium text-[#22211F]">
              {healthState.data?.status || (healthState.loading ? 'probing' : 'offline')}
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3.5">
          <span className="text-[11px] font-medium tracking-wide text-[#8C867C] uppercase">Latency</span>
          <div className="mt-1 flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5 text-[#8C867C]" />
            <span className="font-mono-code text-sm font-medium text-[#22211F]">
              {healthState.latencyMs !== null ? `${healthState.latencyMs} ms` : '—'}
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3.5">
          <span className="text-[11px] font-medium tracking-wide text-[#8C867C] uppercase">Environment</span>
          <div className="mt-1 flex items-center gap-1.5">
            <Globe className="h-3.5 w-3.5 text-[#8C867C]" />
            <span className="font-mono-code text-sm font-medium text-[#22211F]">
              {healthState.data?.environment || 'development'}
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] p-3.5">
          <span className="text-[11px] font-medium tracking-wide text-[#8C867C] uppercase">API Version</span>
          <div className="mt-1 flex items-center gap-1.5">
            <span className="font-mono-code text-sm font-medium text-[#22211F]">
              {healthState.data?.version || 'v0.1.0'}
            </span>
          </div>
        </div>
      </div>

      {/* LLM Provider Status info banner */}
      {healthState.data?.llm && (
        <div className="mt-3 flex flex-wrap items-center justify-between rounded-xl border border-[#ECE7DE] bg-[#FAF9F5] px-4 py-2.5 text-xs">
          <div className="flex items-center gap-2">
            <Cpu className="h-3.5 w-3.5 text-[#D97757]" />
            <span className="text-[#6B655B]">LLM Provider:</span>
            <span className="font-semibold text-[#22211F] uppercase">{healthState.data.llm.provider}</span>
            <span className="text-[#8C867C]">({healthState.data.llm.model})</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                healthState.data.llm.is_reachable && healthState.data.llm.is_model_installed
                  ? 'bg-emerald-500'
                  : 'bg-rose-500'
              }`}
            />
            <span className="font-mono-code text-[11px] text-[#47443E]">
              {healthState.data.llm.is_reachable
                ? healthState.data.llm.is_model_installed
                  ? 'Model Ready'
                  : 'Model Not Installed'
                : 'Service Offline'}
            </span>
          </div>
        </div>
      )}

      {/* If offline, offer helpful start instructions */}
      {!isHealthy && !healthState.loading && (
        <div className="mt-4 rounded-xl border border-[#EBD6CF] bg-[#FDF4F2] p-4 text-xs text-[#7A362B]">
          <div className="flex items-center gap-2 font-medium">
            <Terminal className="h-4 w-4" />
            <span>FastAPI server is not detected on http://localhost:8000</span>
          </div>
          <p className="mt-1.5 text-[#8F4335]">
            Run the following in PowerShell from the <code className="rounded bg-[#F8E2DC] px-1 py-0.5 font-mono-code">backend/</code> directory to start the dev server:
          </p>
          <div className="mt-2 flex items-center justify-between rounded-lg bg-[#24211E] px-3 py-2 text-[#EFEBE4]">
            <code className="font-mono-code text-[11px]">.venv\Scripts\uvicorn app.main:app --reload --port 8000</code>
          </div>
        </div>
      )}

      {/* JSON Viewer */}
      {showJson && (
        <div className="mt-4 overflow-hidden rounded-xl border border-[#E4DECFA0] bg-[#22201D] p-4">
          <div className="flex items-center justify-between pb-2 text-[11px] text-[#A69F91]">
            <span className="font-mono-code">Response Payload</span>
            <span>{healthState.lastChecked ? healthState.lastChecked.toLocaleTimeString() : ''}</span>
          </div>
          <pre className="overflow-x-auto font-mono-code text-xs text-[#E8E4DA]">
            {JSON.stringify(
              healthState.data || {
                error: healthState.error,
                target: 'http://localhost:8000/api/health',
              },
              null,
              2
            )}
          </pre>
        </div>
      )}
    </div>
  );
}
