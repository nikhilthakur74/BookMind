import {
  BookOpen,
  ExternalLink,
  RefreshCw,
  BarChart2,
  Activity,
  MessageSquare,
  Menu,
} from 'lucide-react';
import type { HealthState } from '../types/api';

export type NavTab = 'chat' | 'evaluation' | 'health';

interface HeaderProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  healthState: HealthState;
  onRefresh: () => void;
  onOpenMobileSidebar?: () => void;
}

export function Header({
  activeTab,
  onSelectTab,
  healthState,
  onRefresh,
  onOpenMobileSidebar,
}: HeaderProps) {
  const isHealthy = healthState.data?.status === 'healthy';

  return (
    <header className="sticky top-0 z-40 w-full border-b border-[#ECE7DE] bg-[#FAF9F5]/90 backdrop-blur-md transition-all">
      <div className="flex h-15 items-center justify-between px-4 sm:px-6">
        {/* Brand identity & mobile menu toggle */}
        <div className="flex items-center gap-3">
          {onOpenMobileSidebar && (
            <button
              onClick={onOpenMobileSidebar}
              className="lg:hidden p-1.5 rounded-lg text-[#6E6960] hover:text-[#22211F] hover:bg-[#EFEAE1] transition-colors"
              title="Toggle sidebar"
            >
              <Menu className="w-5 h-5" />
            </button>
          )}

          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#272522] text-[#FAF9F5] shadow-xs">
              <BookOpen className="h-4 w-4 text-[#E6A07E]" strokeWidth={2.2} />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-serif-claude text-lg sm:text-xl font-medium tracking-tight text-[#22211F]">
                BookMind
              </span>
              <span className="hidden text-[10px] font-medium tracking-wider uppercase text-[#88837A] md:inline-block">
                Deep Reading & RAG
              </span>
            </div>
          </div>
        </div>

        {/* Center navigation tabs */}
        <nav className="flex items-center rounded-xl bg-[#EFEAE1]/70 p-1 border border-[#E2DDD3]">
          <button
            onClick={() => onSelectTab('chat')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'chat'
                ? 'bg-[#FAF9F5] text-[#22211F] shadow-xs'
                : 'text-[#6E6960] hover:text-[#22211F]'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Reader & Chat</span>
          </button>

          <button
            onClick={() => onSelectTab('evaluation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'evaluation'
                ? 'bg-[#FAF9F5] text-[#22211F] shadow-xs'
                : 'text-[#6E6960] hover:text-[#22211F]'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>Evaluation</span>
          </button>

          <button
            onClick={() => onSelectTab('health')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'health'
                ? 'bg-[#FAF9F5] text-[#22211F] shadow-xs'
                : 'text-[#6E6960] hover:text-[#22211F]'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Diagnostics</span>
          </button>
        </nav>

        {/* Right side live status & docs */}
        <div className="flex items-center gap-2.5">
          {/* Health indicator pill */}
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] transition-colors ${
              healthState.loading
                ? 'border-[#E2DDCF] bg-[#F3EFE6] text-[#7A756C]'
                : isHealthy
                ? 'border-[#CFE2CE] bg-[#EDF7EC] text-[#2F6830]'
                : 'border-[#ECD0CC] bg-[#FDF1F0] text-[#9A3428]'
            }`}
          >
            <span className="relative flex h-2 w-2">
              {healthState.loading ? (
                <span className="inline-flex h-2 w-2 animate-ping rounded-full bg-amber-400 opacity-75" />
              ) : isHealthy ? (
                <>
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                </>
              ) : (
                <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
              )}
            </span>
            <span className="font-medium hidden sm:inline">
              {healthState.loading
                ? 'Checking...'
                : isHealthy
                ? `Online (${healthState.latencyMs}ms)`
                : 'Offline'}
            </span>
            <button
              onClick={onRefresh}
              disabled={healthState.loading}
              title="Refresh health check"
              className="rounded p-0.5 text-current hover:opacity-75 disabled:opacity-40"
            >
              <RefreshCw className={`h-2.5 w-2.5 ${healthState.loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Docs link */}
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-1 rounded-lg border border-[#E5E0D5] bg-[#FAF9F5] px-2 py-1 text-[11px] font-medium text-[#4A4741] transition-all hover:bg-[#F2EEE4] hover:text-[#22211F] md:inline-flex"
          >
            <span>FastAPI Docs</span>
            <ExternalLink className="h-3 w-3 opacity-60" />
          </a>
        </div>
      </div>
    </header>
  );
}
